from typing import Type

from torch_geometric import transforms
from torch_geometric.transforms import BaseTransform

from readouts.utils.utils import import_from_string

from .custom_virtual_node import CustomVirtualNode
from .initialization import RandomNormalNodeInitialization
from .shape_transforms import LabelShapeAdjust, OneHotEncoder

__all__ = [
    "CustomVirtualNode",
    "LabelShapeAdjust",
    "OneHotEncoder",
    "RandomNormalNodeInitialization",
]
# ensure no name overlap with torch_geometric transforms
assert all(trf_name not in transforms.__all__ for trf_name in __all__)
__all__ += transforms.__all__


def get_transform(transform_name: str) -> Type[BaseTransform]:
    """Return class of transform , whether from our implementation or torch geometric."""
    try:
        return import_from_string(f"readouts.transforms.{transform_name}")
    except ImportError:
        return import_from_string(f"torch_geometric.transforms.{transform_name}")
