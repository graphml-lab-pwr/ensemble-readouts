import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomNormalNodeInitialization(BaseTransform):
    """Initializes features of data with random normal distribution, only when x == None in data."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, data: Data) -> Data:
        assert data.x is None
        data.x = torch.randn((len(data.y), self.dim))
        return data
