from typing import Dict, Optional

import torch
from torch.types import Number
from torchmetrics import Accuracy
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric
from torchmetrics import PearsonCorrCoef
from torchmetrics import Perplexity
from torchmetrics import R2Score
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)
from torchmetrics.functional.classification.specificity import _specificity_reduce


class MetricsDict(torch.nn.ModuleDict):
    def __init__(self, **metrics: Metric) -> None:
        """
        An alternative to `torchmetrics.MetricCollection` that is more explicit.
        It does not define an `update` method. Rather, the user should explicitly call
        `update` on the dict's values.
        """
        super().__init__(metrics)

    def compute(self) -> Dict[str, Number]:
        """
        Calls `compute` in a loop over all the metrics in the dict.
        Returns
        -------
        Dict[str, Number]
            The computed metric values.
        """
        return {
            metric_name: metric.compute().item() for metric_name, metric in self.items()
        }

    def reset(self) -> None:
        """
        Calls `reset` in a loop over all metrics in the dict.
        """
        for metric in self.values():
            metric.reset()

    def __add__(self, other: "MetricsDict") -> "MetricsDict":
        """
        Merges two MetricsDicts.
        """
        return MetricsDict(**self, **other)


class BinaryBalancedAccuracy(BinaryStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> torch.Tensor:
        tp, fp, tn, fn = self._final_state()
        binary_specificity = _specificity_reduce(
            tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )
        binary_recall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
        )
        return (binary_specificity + binary_recall) * 0.5


def get_regression_metrics(prefix: str = "") -> MetricsDict:
    """
    Returns a MetricsDict for regression tasks.
    Args:
        prefix: A prefix to be added to the metric names.
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    return MetricsDict(
        **{
            f"{prefix}mae": MeanAbsoluteError(),
            f"{prefix}rmse": MeanSquaredError(squared=False),
            f"{prefix}r_squared": R2Score(),
            f"{prefix}pearson_corr_coef": PearsonCorrCoef(),
        }
    )


def get_mlm_metrics(
    prefix: str = "", num_classes: int = -1, ignore_index: int = -100
) -> MetricsDict:
    """
    Returns a MetricsDict for MLM tasks.
    Args:
        prefix: A prefix to be added to the metric names.
        num_classes: The number of classes in the vocabulary.
        ignore_index: The index to be ignored in the calculation.
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    return MetricsDict(
        **{
            f"{prefix}accuracy": Accuracy(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=ignore_index,
                top_k=1,
            ),
            f"{prefix}accuracy_top5": Accuracy(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=ignore_index,
                top_k=5,
            ),
            f"{prefix}perplexity": Perplexity(ignore_index=ignore_index),
        }
    )


def get_classification_metrics() -> MetricsDict:
    return MetricsDict(accuracy=Accuracy(task="binary"))
