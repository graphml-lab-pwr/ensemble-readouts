import importlib
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import LightningDataset


def parent_name_impl(_parent_) -> str:
    """Custom resolver for parent key name interpolation.
    Credit: https://github.com/omry/omegaconf/discussions/937#discussioncomment-2787746
    """
    return _parent_._key()


OmegaConf.register_new_resolver("parent_name", parent_name_impl)


def load_config(
    path: str | Path, config_key: str | None = None, key_store_name: str | None = None
) -> dict[Any, Any]:
    config = OmegaConf.load(path)
    assert isinstance(config, DictConfig)
    config_primitive = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config_primitive, dict)

    if config_key:
        config_primitive = config_primitive[config_key]

    if key_store_name:
        config_primitive[key_store_name] = config_key

    return config_primitive


def import_from_string(dotted_path: str) -> Any:
    """
    The function taken from github.com/UKPLab/sentence-transformers.
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except ModuleNotFoundError:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)


def get_train_labels_loss_weights(datamodule: LightningDataset, output_dim: int) -> torch.Tensor:
    """Function infers label counts for the train dataset."""
    class_count = torch.bincount(datamodule.train_dataset.data.y.flatten())
    weights = (1 - class_count / class_count.sum()).tolist()

    # for BCEWithLogitLoss setup pos_weight
    if output_dim == 1:
        weights = weights[1]

    return weights
