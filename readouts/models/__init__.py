from typing import Type

from readouts.utils.utils import import_from_string

from .model_base import ModelBase
from .model_graph import GraphModel, GraphModelConfig

__all__ = [
    "GraphModelConfig",
    "GraphModel",
    "ModelBase",
    "get_model_cls",
]


def get_model_cls(model_name: str) -> Type[ModelBase]:
    return import_from_string(f"readouts.models.{model_name}")
