from abc import ABCMeta
from abc import abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

from mole.metrics import MetricsDict

# Define type alias for configuration dictionaries
OptimizerConfig = Dict[str, Any]
SchedulerConfig = Dict[str, Any]

# Type alias for PTL configure_optimizers return value
ConfiguredOptimizers = Tuple[
    List[torch.optim.Optimizer],
    Optional[List[Dict[str, Union[str, torch.optim.lr_scheduler._LRScheduler]]]],
]  # noqa: E231
TensorDict = Dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


class Model(pl.LightningModule, metaclass=ABCMeta):
    """
    Defines an interface that provides:
    - appropriate, scalable metric computation
    """

    def __init__(
        self,
        # Changed optimizer and scheduler args to expect config dicts
        optimizer_cfg: OptimizerConfig,
        metrics: MetricsDict,
        scheduler_cfg: Optional[SchedulerConfig] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Base class for MolE models, subclassed from PyTorch Lightning LightningModule.

        Args:
            optimizer_cfg: Dictionary containing configuration for the optimizer
                           (e.g., {'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.01}).
            metrics: Dictionary of metrics.
            scheduler_cfg: Optional dictionary containing configuration for the LR scheduler
                           (e.g., {'scheduler_type': 'linear_warmup', 'num_warmup_steps': 100}).
            checkpoint_path: Optional path to a checkpoint for fine-tuning.
        """
        super().__init__()
        # Store the config dicts instead of partials
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg
        self.metrics = metrics
        self.checkpoint_path = checkpoint_path
        # Placeholder for the actual optimizer/scheduler instances created later
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def prepare_data(self) -> None:
        # Download checkpoints for fine-tuning should be in here.
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        """Base behavior: load state dict from fine-tune ckpt, if applicable."""
        if self.checkpoint_path is not None:
            # Add check if checkpoint file exists
            if not os.path.exists(self.checkpoint_path):
                logger.warning(
                    f"Checkpoint path specified but not found: {self.checkpoint_path}"
                )
                # Decide how to handle - error out or continue with random init?
                # For now, log warning and proceed (model will be randomly initialized)
            else:
                try:
                    state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                    # Simplify state_dict loading logic assuming standard PTL format
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    else:
                        # Handle cases where checkpoint might just be the state_dict itself
                        logger.info(
                            "'state_dict' key not found in checkpoint, assuming checkpoint file *is* the state_dict."
                        )

                    # Remove potential prefix (e.g., 'model.') added by Lightning
                    state_dict = {
                        k.partition("model.")[-1]: v for k, v in state_dict.items()
                    }

                    # Handle potential mismatches in prediction head size during fine-tuning
                    # (Example logic, might need adjustment based on actual head structure)
                    head_prefix = "prediction_head."
                    current_head_keys = {
                        k for k in self.state_dict() if k.startswith(head_prefix)
                    }
                    ckpt_head_keys = {
                        k for k in state_dict if k.startswith(head_prefix)
                    }

                    if ckpt_head_keys and current_head_keys:
                        # Simple check: if output layer dimensions mismatch, remove head weights from checkpoint
                        output_layer_key_part = (
                            ".classifier.bias"  # Example, adjust if layer name differs
                        )
                        ckpt_output_key = next(
                            (
                                k
                                for k in ckpt_head_keys
                                if k.endswith(output_layer_key_part)
                            ),
                            None,
                        )
                        current_output_key = next(
                            (
                                k
                                for k in current_head_keys
                                if k.endswith(output_layer_key_part)
                            ),
                            None,
                        )

                        if ckpt_output_key and current_output_key:
                            if (
                                state_dict[ckpt_output_key].shape
                                != self.state_dict()[current_output_key].shape
                            ):
                                logger.warning(
                                    f"Prediction head shape mismatch for {self.checkpoint_path}. "
                                    f"Removing head weights. Fine-tuning will train a new head."
                                )
                                keys_to_remove = [
                                    k for k in state_dict if k.startswith(head_prefix)
                                ]
                                for k in keys_to_remove:
                                    del state_dict[k]

                    load_result = self.load_state_dict(state_dict, strict=False)
                    logger.info(
                        f"Loaded state_dict from {self.checkpoint_path}. Load result: {load_result}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load state_dict from {self.checkpoint_path}: {e}"
                    )
                    # Decide how to handle - error out or continue with random init?
                    # For now, log error and proceed

        # set padding embeddings to 0 for training
        # This part assumes a specific model structure ('self.model.MolE...').
        # It should ideally be handled within the specific model's setup or init.
        # Commenting out here as it's too specific for the base class.
        # if stage == "train":
        #     with torch.no_grad():
        #         # self.model.MolE.encoder.rel_embeddings.weight[0].fill_(0)
        #         # self.model.MolE.embeddings.word_embeddings.weight[
        #         #     getattr(self.model.config, "padding_idx", 0)
        #         # ].fill_(0)
        #         pass # Model-specific setup should handle this

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Union[torch.optim.Optimizer, ConfiguredOptimizers]:
        """Constructs optimizer and learning rate scheduler from stored config dicts."""
        # --- Create Optimizer ---
        optimizer_name = self._optimizer_cfg.get("optimizer", "AdamW").lower()
        lr = self._optimizer_cfg.get("lr")
        weight_decay = self._optimizer_cfg.get("weight_decay", 0.01)
        if lr is None:
            raise ValueError("'lr' must be specified in optimizer_cfg")

        # Add more optimizers as needed
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )  # Adam has wd but often set to 0
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        if self._scheduler_cfg is None:
            return self.optimizer
        else:
            # --- Create Scheduler ---
            scheduler_type = self._scheduler_cfg.get("scheduler_type", "none").lower()
            num_warmup_steps = self._scheduler_cfg.get("num_warmup_steps", 0)
            num_training_steps = self._scheduler_cfg.get(
                "num_training_steps"
            )  # Required for linear decay

            if scheduler_type == "none":
                return self.optimizer  # No scheduler
            elif scheduler_type == "constant_warmup":
                from mole.engine.schedulers import (
                    get_constant_schedule_with_warmup,
                )  # Import locally

                self.lr_scheduler = get_constant_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=num_warmup_steps
                )
            elif scheduler_type == "linear_warmup":
                if num_training_steps is None:
                    raise ValueError(
                        "'num_training_steps' must be provided in scheduler_cfg for linear_warmup scheduler."
                    )
                from mole.engine.schedulers import (
                    get_linear_schedule_with_warmup,
                )  # Import locally

                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

            # PTL pattern
            return [self.optimizer], [
                {
                    "scheduler": self.lr_scheduler,
                    "interval": "step",  # Default interval to step
                }
            ]

    @abstractmethod
    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        """
        Update your metric states here. This will be called for every training and validation batch.
        See `photosynthetic.training.core.metrics.MetricsDict` for more info.

        At the end of each training step, metrics will be computed globally, logged, and reset.
        For validation, metrics are accumulated on each rank over all validation batches. At the end of
        the validation epoch, they will be computed globally, logged, and reset.
        """
        pass

    def on_train_batch_end(  # type: ignore[override]
        self,
        outputs: TensorDict,
        batch: TensorDict,
        batch_idx: int,
    ) -> None:
        """
        1. Calls `update_metrics`
        2. Synchronizes / computes metrics across all ranks
        3. Logs the metric values
        4. Resets the metric states
        """
        self.update_metrics(outputs=outputs, batch=batch)
        self.log_dict({"train_" + k: v for k, v in self.metrics.compute().items()})
        self.metrics.reset()

    def on_validation_batch_end(  # type: ignore[override]
        self,
        outputs: TensorDict,
        batch: TensorDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Calls `update_metrics`.
        """
        self.update_metrics(outputs=outputs, batch=batch)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> TensorDict:
        # Assuming validation_step returns the desired output dict for prediction
        step_output = self.validation_step(batch=batch, batch_idx=batch_idx)  # type: ignore[misc]
        # Ensure predict_step returns a dictionary, even if validation_step doesn't
        if isinstance(
            step_output, torch.Tensor
        ):  # e.g. if validation_step just returns loss
            # Need to define what predict_step should return. Maybe run forward pass again?
            # For now, return an empty dict or raise error, as behavior is undefined.
            logger.warning(
                "Predict step received Tensor from validation_step, returning empty dict."
            )
            return {}
        return step_output if step_output is not None else {}

    def on_validation_epoch_end(self) -> None:
        """
        1. Synchronizes / computes metrics across all ranks
        2. Logs the metric values
        3. Resets the metric states
        """
        self.log_dict({"val_" + k: v for k, v in self.metrics.compute().items()})
        self.metrics.reset()

    def export_onnx(self, filepath: str) -> None:
        # Add check if self.model exists, as it's not initialized in base __init__
        if not hasattr(self, "model") or not isinstance(self.model, nn.Module):
            logger.error(
                "Cannot export ONNX: self.model not found or not an nn.Module."
            )
            return

        filepath = filepath.replace(".ckpt", ".onnx")
        # Adjust path replacement if needed
        filepath = filepath.replace("checkpoints", "onnx_models")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        output_names = ["output"]
        # Infer input names from the model's forward signature if possible, or keep defaults
        input_names = [
            "input_ids",
            "input_mask",
            "relative_pos",
        ]  # Default if model forward isn't inspectable easily

        # Create realistic input sample based on potential model config (if available)
        # Defaults used here might need adjustment
        bs = 1
        seq_len = 64  # Example sequence length
        try:
            input_sample = {
                "input_ids": torch.randint(
                    0,
                    getattr(self.model.config, "vocab_size", 100),
                    (bs, seq_len),
                    dtype=torch.long,
                ),
                "input_mask": torch.ones((bs, seq_len), dtype=torch.bool),
                "relative_pos": torch.randint(
                    0,
                    getattr(self.model.config, "max_relative_positions", 5) * 2,
                    (bs, seq_len, seq_len),
                    dtype=torch.long,
                ),
            }
        except AttributeError:
            logger.warning(
                "Could not get model config for ONNX input sample generation. Using generic defaults."
            )
            input_sample = {
                "input_ids": torch.ones((bs, seq_len), dtype=torch.long),
                "input_mask": torch.ones((bs, seq_len), dtype=torch.bool),
                "relative_pos": torch.ones((bs, seq_len, seq_len), dtype=torch.long),
            }

        dynamic_axes_dict = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "input_mask": {0: "batch_size", 1: "sequence_length"},
            "relative_pos": {
                0: "batch_size",
                1: "sequence_length",
                2: "sequence_length",
            },
            "output": {0: "batch_size"},
        }
        try:
            self.to_onnx(
                filepath,
                input_sample,
                export_params=True,
                opset_version=11,  # Specify opset version for compatibility
                dynamic_axes=dynamic_axes_dict,
                input_names=input_names,
                output_names=output_names,
                verbose=False,
            )
            logger.info(f"Model exported to ONNX format at {filepath}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
