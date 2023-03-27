from collections import OrderedDict

import torch
from torch import nn as nn
from torch.nn import GRU
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from readouts.models.readout_layers import ReadoutLayer
from readouts.models.readout_layers.output import ReadoutOutput


class DenseReadout(ReadoutLayer):
    """Non-invariant MLP readout as proposed in https://arxiv.org/abs/2211.04952

    Concatenates representation of each node in a graph to one big vector, eventually padded to
    maximum possible number of nodes.

    Credit: https://github.com/davidbuterez/gnn-neural-readouts
    """

    def __init__(
        self,
        max_num_nodes_in_graph: int,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.max_num_nodes_in_graph = max_num_nodes_in_graph
        self.dense_input_dim = max_num_nodes_in_graph * input_dim
        self.layer_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear",
                        nn.Linear(
                            in_features=self.dense_input_dim,
                            out_features=intermediate_dim,
                        ),
                    ),
                    ("batch_norm", nn.BatchNorm1d(intermediate_dim)),
                    ("activation", nn.ReLU()),
                ]
            )
        )
        self.layer_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear",
                        nn.Linear(in_features=intermediate_dim, out_features=output_dim),
                    ),
                    ("batch_norm", nn.BatchNorm1d(output_dim)),
                    ("activation", nn.ReLU()),
                    ("dropout", nn.Dropout(p=dropout_rate)),
                ]
            )
        )

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        graph_repr = self.forward_repr(node_reprs, batch)["output"]
        return ReadoutOutput(graph_reprs=graph_repr)

    def forward_repr(
        self, node_reprs_padded: torch.Tensor, batch: Batch
    ) -> dict[str, torch.Tensor]:
        """Auxiliary function for obtaining intermediate representations."""
        node_reprs_padded, _ = to_dense_batch(
            node_reprs_padded,
            batch.batch,
            fill_value=0.0,
            max_num_nodes=self.max_num_nodes_in_graph,
        )
        x = node_reprs_padded.view(-1, self.dense_input_dim)
        layer_1_out = self.layer_1(x)
        layer_2_out = self.layer_2(layer_1_out)
        return {"layer_1_out": layer_1_out, "output": layer_2_out}


class GRUReadout(ReadoutLayer):
    """GRU rnn layer as proposed in https://arxiv.org/abs/2211.04952.
    Credit:
        https://github.com/davidbuterez/gnn-neural-readouts/blob/main/code/models/graph_models.py

    Note: This model works feature-wise, i.e., at each time-step i-th node features from all
    node are fed to the RNN.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_num_nodes_in_graph: int,
        bidirectional=False,
        num_layers=1,
    ):
        super().__init__()
        self.max_num_nodes_in_graph = max_num_nodes_in_graph
        self.gru_agg = GRU(
            batch_first=True,
            input_size=max_num_nodes_in_graph,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

    def forward(self, node_reprs: torch.Tensor, batch: Batch) -> ReadoutOutput:
        node_reprs_padded, _ = to_dense_batch(
            node_reprs, batch.batch, fill_value=0, max_num_nodes=self.max_num_nodes_in_graph
        )
        output, states = self.gru_agg(node_reprs_padded.permute(0, 2, 1))
        graph_reprs = output[:, -1, :]
        return ReadoutOutput(graph_reprs=graph_reprs)
