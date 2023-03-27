from pathlib import Path
from typing import Any, Callable

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import LightningDataset, LightningNodeData
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.transforms import Compose

from readouts.transforms import LabelShapeAdjust, OneHotEncoder, get_transform

VAL_FRAC = 0.1
TEST_FRAC = 0.1


def get_benchmark_datamodule(
    root_dir: str | Path,
    task_level: str,
    dataset_type: str,
    dataset_name: str,
    output_dim: int,
    batch_size: int,
    num_workers: int,
    transforms_config: dict[str, dict[str, Any]] | None = None,
    **dataset_kwargs,
) -> LightningDataset | LightningNodeData:
    """Prepares ready-to-train lightning datamodule with train/val/test splits."""
    if task_level == "graph":
        return get_graph_level_benchmark_datamodule(
            root_dir=root_dir,
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            output_dim=output_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            transforms_config=transforms_config,
            **dataset_kwargs,
        )
    else:
        raise ValueError(f"Invalid task_level: {task_level}")


def get_graph_level_benchmark_datamodule(
    root_dir: str | Path,
    dataset_type: str,
    dataset_name: str,
    output_dim: int,
    batch_size: int,
    num_workers: int,
    transforms_config: dict[str, dict[str, Any]] | None = None,
    **dataset_kwargs,
) -> LightningDataset:
    """Loads benchmark dataset based on the given name."""
    transforms: list[Callable] = []

    if transforms_config:
        for trf_name, trf_args in transforms_config.items():
            trf_cls = get_transform(trf_name)
            trf = trf_cls(**trf_args)
            transforms.append(trf)

    if dataset_type == "tud":
        dataset = TUDataset(
            root_dir,
            dataset_name,
            transform=Compose([LabelShapeAdjust(output_dim == 1)] + transforms),
            **dataset_kwargs,
        )
        train_idx, val_idx, test_idx = get_split_idx(len(dataset))
        train_ds = dataset[train_idx]
        val_ds = dataset[val_idx]
        test_ds = dataset[test_idx]
    elif dataset_type == "zinc":
        composed_transforms = Compose(
            [OneHotEncoder(num_classes=28), LabelShapeAdjust(output_dim == 1)] + transforms
        )
        train_ds = ZINC(root_dir, split="train", transform=composed_transforms, **dataset_kwargs)
        val_ds = ZINC(root_dir, split="val", transform=composed_transforms, **dataset_kwargs)
        test_ds = ZINC(root_dir, split="test", transform=composed_transforms, **dataset_kwargs)
    else:
        raise ValueError(f"Invalid dataset name or not implemented yet: {dataset_type}")

    return LightningDataset(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_split_idx(
    dataset_size: int,
    stratify: torch.Tensor | None = None,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.arange(dataset_size)
    val_test_frac = val_frac + test_frac
    train_idx, val_test_idx = train_test_split(index, test_size=val_test_frac, stratify=stratify)
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=test_frac / val_test_frac, stratify=stratify
    )

    assert not set(train_idx).intersection(set(val_idx))
    assert not set(train_idx).intersection(set(test_idx))
    assert not set(val_idx).intersection(set(test_idx))

    return train_idx, val_idx, test_idx
