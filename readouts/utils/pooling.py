"""
The following functions implements aggregations which are not available in torch_scatter for now.
"""

from functools import partial
from typing import Callable

import torch


def global_median_pool(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Computes median aggregation of x according to the index with median along 1 dim."""
    return _global_pool(x, index, _median)


def global_std_pool(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Computes median aggregation of x according to the index with std along 1 dim."""
    feature_wise_std = partial(torch.std, dim=1)
    return _global_pool(x, index, feature_wise_std)


def _global_pool(
    x: torch.Tensor, index: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Computes aggregation of x according to the index with a given function."""
    _, feat_dim = x.shape
    num_items = int(torch.max(index).item()) + 1
    aggregated_x = torch.empty(num_items, feat_dim, device=x.device, dtype=x.dtype)

    for i in range(num_items):
        x_for_index = x[index == i]
        assert len(x_for_index)
        aggregated_x[i] = func(x_for_index)

    return aggregated_x


def _median(t: torch.Tensor) -> torch.Tensor:
    return torch.median(t, dim=0).values
