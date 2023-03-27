from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool

from readouts.models.metrics import activations
from readouts.models.readout_layers.output import ReadoutOutput
from readouts.utils.pooling import global_median_pool, global_std_pool


class ReadoutLayer(nn.Module):
    """Base class for readout (pooling) methods."""

    @abstractmethod
    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        """Takes batch-of-graphs node representations, where each tensor refers to one graph."""
        raise NotImplementedError


class AggregateReadoutBaseLayer(ReadoutLayer, ABC):
    """Naive readout returning simple aggregation of nodes' embeddings."""

    def __init__(self, agg_func_name: str):
        super().__init__()
        self.agg = self.get_agg(agg_func_name)

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        return ReadoutOutput(graph_reprs=self.agg(node_reprs, batch.batch))

    @staticmethod
    def get_agg(agg_func_name: str) -> Callable[[torch.Tensor, torch.LongTensor], torch.Tensor]:
        if agg_func_name == "mean":
            return global_mean_pool
        elif agg_func_name == "sum":
            return global_add_pool
        elif agg_func_name == "max":
            return global_max_pool
        elif agg_func_name == "median":
            return global_median_pool
        elif agg_func_name == "std":
            return global_std_pool
        else:
            raise ValueError(f"Invalid agg_name: {agg_func_name}")


class AggregateReadoutMax(AggregateReadoutBaseLayer):
    def __init__(self):
        super().__init__("max")


class AggregateReadoutMean(AggregateReadoutBaseLayer):
    def __init__(self):
        super().__init__("mean")


class AggregateReadoutSum(AggregateReadoutBaseLayer):
    def __init__(self):
        super().__init__("sum")


class AggregateReadoutMedian(AggregateReadoutBaseLayer):
    def __init__(self):
        super().__init__("median")


class AggregateReadoutStd(AggregateReadoutBaseLayer):
    def __init__(self):
        super().__init__("std")


class VirtualNodeReadout(ReadoutLayer):
    """Readout returning representation of a virtual node as graph representation."""

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        return ReadoutOutput(graph_reprs=node_reprs[batch.ptr[1:] - 1])


class DeepSetsBase(ReadoutLayer):
    """Weighting of representation with sum readout, resembles Dense readout.

    In combination with MLP at predictor layer, it forms exactly DeepSets.

    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        dropout_rate: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.node_transform_mlp = self.get_mlp(
            input_dim, intermediate_dim, output_dim, activation, dropout_rate
        )

    def get_mlp(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        dropout_rate: float,
    ) -> nn.Module:
        act_func = activations[activation]
        return nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            act_func(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        node_reprs = self.node_transform_mlp(node_reprs)
        graph_reprs = global_add_pool(node_reprs, batch.batch)

        return ReadoutOutput(graph_reprs=graph_reprs)


class DeepSetsLarge(DeepSetsBase):
    """Large version of DeepSetsBase."""

    def get_mlp(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: str,
        dropout_rate: float,
    ) -> nn.Module:
        act_func = activations[activation]
        return nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            act_func(),
            nn.Linear(intermediate_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            act_func(),
            nn.Dropout(p=dropout_rate),
        )
