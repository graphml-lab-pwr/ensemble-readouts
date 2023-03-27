from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import torch
import torch.nn as nn
from pydantic import BaseModel
from torchmetrics import MetricCollection

T_forward = TypeVar("T_forward")


class PredictorBase(nn.Module, ABC, Generic[T_forward]):
    config_cls: Type[BaseModel] | None = None

    def __init__(self, loss: nn.Module, metrics: MetricCollection):
        super().__init__()
        self.loss = loss
        self.metrics = metrics

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> T_forward:
        """Main body of forward pass for the concrete predictor."""
        return NotImplemented

    def logits_to_preds(self, x: T_forward) -> torch.Tensor:
        """Allows to obtain final prediction from the output of forward (by default returns same)"""
        assert isinstance(x, torch.Tensor)
        return x
