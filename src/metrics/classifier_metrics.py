from typing import Any, Literal

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    JaccardIndex,
    Precision,
    Recall,
)


class ClassifierMetric:
    """Give custom metric collection, it has update, compute and clone interface"""

    def __init__(
        self,
        num_classes: int,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        top_k: int | None = None,
    ) -> None:
        self.num_classes = num_classes

        if not top_k:
            top_k = 1

        metrics = [
            Accuracy(task="multiclass", num_classes=num_classes, average=average, top_k=top_k),
            Precision(task="multiclass", num_classes=num_classes, average=average, top_k=top_k),
            Recall(task="multiclass", num_classes=num_classes, average=average, top_k=top_k),
            F1Score(task="multiclass", num_classes=num_classes, average=average, top_k=top_k),
            JaccardIndex(task="multiclass", num_classes=num_classes, average=average),
        ]

        self.metrics = MetricCollection(metrics)

    def update(self, predictions: torch.Tensor, target: torch.Tensor):
        probs = torch.softmax(predictions, dim=1)[:, 1]
        one_hot_target: torch.Tensor = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        self.metrics.update(probs, one_hot_target)

    def compute(self) -> dict[str, Any]:
        return self.metrics.compute()

    def clone(self, prefix: None | str = None, postfix: None | str = None) -> MetricCollection:
        return self.metrics.clone(prefix, postfix)

    def reset(self):
        self.metrics.reset()
