from abc import ABC

import torch
from torch import nn as nn

from readouts.models.predictors import MLPClassifier, MLPRegression, PredictorBase


class EnsembleDecisionModel(PredictorBase, ABC):
    """Base class for all ensemble models."""

    def __init__(self, predictor: PredictorBase):
        super().__init__(predictor.loss, predictor.metrics)
        self.predictor = predictor


class EnsembleMeanDecision(EnsembleDecisionModel):
    """Ensemble averaging outputs of a predictor.

    Performs mean aggregation over first dimension (e.g. representation from several readouts).

    """

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        logits = self.predictor(x, *args, **kwargs)
        assert len(logits.shape) == 3
        return torch.mean(logits, dim=1)


class EnsembleWeightedDecision(EnsembleDecisionModel):
    """Ensemble with learnable weighted sum of outputs of a predictor.

    Performs sum aggregation over first dimension (e.g. representation from several readouts),

    """

    def __init__(self, predictor: PredictorBase, num_readouts: int):
        super().__init__(predictor)
        self.ensemble_transform = nn.Sequential(nn.Linear(num_readouts, 1))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        logits = self.predictor(x, *args, **kwargs)
        return self.ensemble_transform(logits.permute(0, 2, 1)).squeeze(-1)


class EnsembleProjectedWeightedDecision(EnsembleWeightedDecision):
    """Ensemble model with projection, and learnable weighted sum of outputs of a predictor.

    Projection applied separately for each data in first dimension.
    Performs sum aggregation over first dimension (e.g. representation from several readouts),

    """

    def __init__(self, predictor: PredictorBase, num_readouts: int, input_dim: int):
        super().__init__(predictor, num_readouts)
        self.proj_weights = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_readouts)]
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_projected = torch.empty_like(x)
        for i, proj_layer in enumerate(self.proj_weights):
            x_projected[:, i, :] = proj_layer(x[:, i, :])
        return super().forward(x_projected)


class MLPClassifierEnsembleMeanDecision(EnsembleMeanDecision):
    """Classifier for the case when multiple dimension per single data item is present.

    This model should be used along with MultipleReadout (no concat).
    The input should have dimensions: [batch_size, num_readouts, feature_dim]
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        *args,
        **kwargs,
    ):
        predictor = MLPClassifier(
            input_dim, intermediate_dim, output_dim, activation, *args, **kwargs
        )
        super().__init__(predictor)


class MLPClassifierEnsembleWeightedDecision(EnsembleWeightedDecision):
    """Classifier which weights prediction from each representation (readout) dimension.

    This model should be used along with MultipleReadout (no concat).
    The input should have dimensions: [batch_size, num_readouts, feature_dim]

    """

    def __init__(
        self,
        num_readouts: int,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        *args,
        **kwargs,
    ):
        predictor = MLPClassifier(
            input_dim, intermediate_dim, output_dim, activation, *args, **kwargs
        )
        super().__init__(predictor, num_readouts)


class MLPClassifierEnsembleProjectedWeightedDecision(EnsembleProjectedWeightedDecision):
    """Classifier which projects each representation (readout) and weights predictions.

    This model should be used along with MultipleReadout (no concat).
    The input should have dimensions: [batch_size, num_readouts, feature_dim]

    """

    def __init__(
        self,
        num_readouts: int,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        *args,
        **kwargs,
    ):
        predictor = MLPClassifier(
            input_dim, intermediate_dim, output_dim, activation, *args, **kwargs
        )
        super().__init__(predictor, num_readouts, input_dim)


class MLPRegressionEnsembleMeanDecision(EnsembleMeanDecision):
    """Classifier for the case when multiple dimension per single data item is present.

    This model should be used along with MultipleReadout (no concat).
    The input should have dimensions: [batch_size, num_readouts, feature_dim]
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
    ):
        predictor = MLPRegression(input_dim, intermediate_dim, output_dim, activation)
        super().__init__(predictor)


class MLPRegressionEnsembleWeightedDecision(EnsembleWeightedDecision):
    """Classifier which weights prediction from each representation (readout) dimension.

    This model should be used along with MultipleReadout (no concat).
    The input should have dimensions: [batch_size, num_readouts, feature_dim]

    """

    def __init__(
        self,
        num_readouts: int,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
    ):
        predictor = MLPRegression(input_dim, intermediate_dim, output_dim, activation)
        super().__init__(predictor, num_readouts)


class MLPRegressionEnsembleProjectedWeightedDecision(EnsembleProjectedWeightedDecision):
    """Regressor which projects each representation (readout) and weights predictions.

    This model should be used along with MultipleReadout (no concat).
    The input should have dimensions: [batch_size, num_readouts, feature_dim]

    """

    def __init__(
        self,
        num_readouts: int,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
    ):
        predictor = MLPRegression(input_dim, intermediate_dim, output_dim, activation)
        super().__init__(predictor, num_readouts, input_dim)
