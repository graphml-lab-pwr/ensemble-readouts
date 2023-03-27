from typing import Any

import torch
from torch import nn as nn
from torch_geometric.data import Batch

from readouts.models.readout_layers import ReadoutLayer, ReadoutOutput, create_readout


class MultipleReadouts(ReadoutLayer):
    """Groups multiple readouts, returns each readout results in a new dimension.

    The output dimension is: [batch_size, readout_dim, feature_dim]

    """

    def __init__(self, readout_configs: dict[str, dict[str, Any]]):
        """
        :param readout_configs: dictionary with readout names and config for each readout
        """
        super().__init__()
        self.readouts = nn.ModuleList(
            [
                create_readout(readout_name, **readout_init_args)
                for readout_name, readout_init_args in readout_configs.items()
            ]
        )

    @property
    def num_readouts(self) -> int:
        return len(self.readouts)

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        readout_reprs: list[torch.Tensor] = []
        for readout_func in self.readouts:
            readout_reprs.append(readout_func(node_reprs, batch).graph_reprs)

        return ReadoutOutput(graph_reprs=torch.stack(readout_reprs, dim=1))


class MultipleReadoutsConcat(MultipleReadouts):
    """Groups multiple readouts, concatenates each readout result into a single vector.

    The output dimension is: [batch_size, readout_dim * feature_dim]

    """

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        readouts_reprs = super().forward(node_reprs, batch).graph_reprs
        _, readout_dim, feature_dim = readouts_reprs.shape
        return ReadoutOutput(graph_reprs=readouts_reprs.reshape(-1, readout_dim * feature_dim))


class MultipleReadoutsWeightedCombine(MultipleReadouts):
    """Combines multiple readouts by weighted sum (+bias) over all readouts.

    The output dimension is: [batch_size, feature_dim]


    """

    def __init__(self, readout_configs: dict[str, dict[str, Any]]):
        super().__init__(readout_configs)
        # note: this is not only weighting but also bias
        self.weights = nn.Linear(self.num_readouts, 1)

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        readouts_reprs = super().forward(node_reprs, batch).graph_reprs
        reprs_combined = self.weights(readouts_reprs.permute(0, 2, 1)).squeeze(-1)
        return ReadoutOutput(graph_reprs=reprs_combined)


class MultipleReadoutsProjectedAndWeightedCombine(MultipleReadouts):
    """Combines multiple readouts by linear projection -> weighted sum (+bias) over all readouts.

    The output dimension is: [batch_size, feature_dim]

    """

    def __init__(self, input_dim: int, readout_configs: dict[str, dict[str, Any]]):
        super().__init__(readout_configs)
        # note: this is not only weighting but also bias
        self.projection_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(self.num_readouts)]
        )
        self.weights = nn.Linear(self.num_readouts, 1)

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        readouts_reprs = super().forward(node_reprs, batch).graph_reprs

        readouts_projected = torch.empty_like(readouts_reprs)
        for i, proj_layer in enumerate(self.projection_layers):
            readouts_projected[:, i, :] = readouts_reprs[:, i, :]

        reprs_combined = self.weights(readouts_projected.permute(0, 2, 1)).squeeze(-1)
        return ReadoutOutput(graph_reprs=reprs_combined)
