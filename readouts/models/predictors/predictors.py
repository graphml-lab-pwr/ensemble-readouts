from abc import ABC

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from readouts.models.metrics import activations, get_classification_metrics, get_regression_metrics
from readouts.models.predictors import PredictorBase


class ClassifierBase(PredictorBase, ABC):
    """Base class for classification models."""

    def __init__(self, output_dim: int, class_weights: list[float] | torch.Tensor | None = None):
        metrics = get_classification_metrics(output_dim)
        weights = torch.as_tensor(class_weights) if class_weights is not None else None
        loss = (
            CrossEntropyLoss(weight=weights)
            if output_dim > 1
            else BCEWithLogitsLoss(pos_weight=weights)
        )
        super().__init__(loss, metrics)
        self.is_binary = output_dim == 1

    def logits_to_preds(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_binary:
            return x.sigmoid()
        return x.softmax(dim=1)


class LogisticRegressionClassifier(ClassifierBase):
    """Simple classification model based on logistic regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        class_weights: list[float] | torch.Tensor | None = None,
    ):
        super().__init__(output_dim, class_weights)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.linear(x)


class MLPClassifier(ClassifierBase):
    """Classification model based on multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        class_weights: list[float] | torch.Tensor | None = None,
    ):
        super().__init__(output_dim, class_weights)
        act_func = activations[activation]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.mlp(x)


class LinearRegression(PredictorBase):
    """Simple linear regression model."""

    def __init__(self, input_dim: int, output_dim: int):
        loss = MSELoss()
        metrics = get_regression_metrics()
        super().__init__(loss, metrics)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.linear(x)


class MLPRegression(PredictorBase):
    """Regression model based on multilayer perceptron."""

    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int, activation: str):
        loss = MSELoss()
        metrics = get_regression_metrics()
        super().__init__(loss, metrics)
        act_func = activations[activation]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.mlp(x)
