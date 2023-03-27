import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MetricCollection, R2Score
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
)

activations = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


def get_classification_metrics(out_channels: int) -> MetricCollection:
    """Returns set of common classification metrics (binary or multiclass) from TorchMetrics."""
    if out_channels > 1:
        return MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(out_channels, average="macro", compute_on_cpu=True),
                "Recall": MulticlassRecall(out_channels, average="macro", compute_on_cpu=True),
                "Precision": MulticlassPrecision(
                    out_channels, average="macro", compute_on_cpu=True
                ),
                "F1Score": MulticlassF1Score(out_channels, average="macro", compute_on_cpu=True),
                "MCC": MulticlassMatthewsCorrCoef(out_channels, compute_on_cpu=True),
                "AUROC": MulticlassAUROC(out_channels, average="macro", compute_on_cpu=True),
            }
        )
    else:
        return MetricCollection(
            {
                "Accuracy": BinaryAccuracy(threshold=0.5, compute_on_cpu=True),
                "Recall": BinaryRecall(threshold=0.5, compute_on_cpu=True),
                "Precision": BinaryPrecision(threshold=0.5, compute_on_cpu=True),
                "F1Score": BinaryF1Score(threshold=0.5, compute_on_cpu=True),
                "MCC": BinaryMatthewsCorrCoef(threshold=0.5, compute_on_cpu=True),
                "AUROC": BinaryAUROC(threshold=0.5, compute_on_cpu=True),
            }
        )


def get_regression_metrics() -> MetricCollection:
    """Returns set of common classification metrics (binary or multiclass) from TorchMetrics."""
    return MetricCollection(
        {
            "MAE": MeanAbsoluteError(),
            "R2": R2Score(),
        }
    )
