from typing import Any

from readouts.models.readout_layers.readouts import ReadoutLayer
from readouts.utils.utils import import_from_string


def create_readout(readout_name: str, **readout_init_args: dict[str, Any]) -> ReadoutLayer:
    """Factory method for building readout layer from its name and config."""
    readout_cls = import_from_string(f"readouts.models.readout_layers.{readout_name}")
    assert issubclass(readout_cls, ReadoutLayer)
    return readout_cls(**readout_init_args)
