from typing import Type

import torch
from torch import nn
from torch_geometric.nn.models import GAT, GCN, GIN
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GraphNeuralNetwork(nn.Module):
    """GNN, optionally with multiple heads for predicting more than one value."""

    def __init__(
        self,
        graph_conv: str,
        input_dim: int,
        hidden_dim: int,
        proj_dim: int | None,
        num_layers: int,
        prediction_heads: int,
        heads_dim: list[int] | None,
        heads_names: list[str] | None = None,
        **conv_kwargs,
    ):
        super().__init__()
        if prediction_heads > 1:
            assert num_layers > 1
            assert heads_dim is not None and len(heads_dim) == prediction_heads
            backbone_num_layers = num_layers - 1
            backbone_proj_dim = None
        else:
            backbone_num_layers = num_layers
            backbone_proj_dim = proj_dim

        if heads_names is not None:
            assert len(heads_names) == prediction_heads
        elif prediction_heads > 1:
            heads_names = [f"head_{i}" for i in range(prediction_heads)]

        assert all(
            basic_kwarg not in conv_kwargs.keys()
            for basic_kwarg in ["in_channels", "hidden_channels", "num_layers", "out_channels"]
        )

        conv_cls = self.get_graph_conv(graph_conv)
        self.graph_conv_backbone = conv_cls(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=backbone_num_layers,
            out_channels=backbone_proj_dim,
            **conv_kwargs,
        )

        if prediction_heads > 1:
            self.conv_heads = nn.ModuleDict()
            assert isinstance(heads_names, list)
            assert isinstance(heads_dim, list)
            for head_name, i_head_dim in zip(heads_names, heads_dim):
                self.conv_heads[head_name] = conv_cls(
                    in_channels=hidden_dim,
                    hidden_channels=-1,  # Layer has dimension (in_channels, out_channels)
                    num_layers=1,
                    out_channels=i_head_dim,
                    **conv_kwargs,
                )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # todo: args, kwargs given for backward compatibility, batch should be passed instead
        x = self.graph_conv_backbone(x, edge_index)

        if hasattr(self, "conv_heads"):
            x_heads: dict[str, torch.Tensor] = {}
            for head_name, head_module in self.conv_heads.items():
                x_heads[head_name] = head_module(x, edge_index)
            return x_heads

        return x

    @staticmethod
    def get_graph_conv(graph_conv: str) -> Type[BasicGNN]:
        if graph_conv == "gcn":
            return GCN
        elif graph_conv == "gin":
            return GIN
        elif graph_conv == "gat":
            return GAT
        else:
            raise ValueError(f"Invalid graph convolution name: {graph_conv}")
