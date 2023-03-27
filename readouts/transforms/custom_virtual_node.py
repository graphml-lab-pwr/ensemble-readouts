import copy
from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose, VirtualNode


class _DropLastEdgeType(BaseTransform):
    """
    The transform removes an edge type with the highest number. This transform can be used to remove
    `out` edges of a virtual node.
    """

    def __call__(self, data: Data) -> Data:
        edge_mask = data["edge_type"] != data["edge_type"].max()
        edge_indices = torch.arange(data.num_edges)[edge_mask]
        old_data = copy.copy(data)
        for key, value in old_data.items():
            if old_data.is_edge_attr(key):
                dim = old_data.__cat_dim__(key, value)
                data[key] = old_data[key].index_select(dim, edge_indices)
        return data


class CustomVirtualNode(BaseTransform):
    def __init__(self, connectivity: Literal["in", "in_out"]):
        """
        :param connectivity: `in_out` adds two new kinds of directed edges (`in` and `out`)
        between graph and a virtual node to allow message passing in both directions. `in` adds only
        one kind of edges to allow message passing only from graph to the virtual node.
        """
        self.config = connectivity

        transform = [VirtualNode()]
        if connectivity == "in":
            transform.append(_DropLastEdgeType())
        elif connectivity != "in_out":
            raise ValueError(f"Wrong connectivity: {connectivity}")
        self.transform = Compose(transform)

    def __call__(self, data: Data) -> Data:
        return self.transform(data)
