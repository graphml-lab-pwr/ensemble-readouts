from abc import ABC
from pathlib import Path
from typing import Any, Literal, Type, TypeVar

from pydantic import BaseModel, Field

from readouts.utils.utils import load_config

T = TypeVar("T", bound="TrainingConfig")


class TrainingConfig(BaseModel, ABC, frozen=True):
    """Base class for all training configurations."""

    experiment_name: str
    model_name: Literal["GraphModel"]
    dataset_dir: Path
    experiment_dir: Path
    wandb_project: str | None
    random_seed: int
    task_level: Literal["graph"]
    dataset_type: str
    dataset_name: str
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)
    input_dim: int
    output_dim: int
    learning_rate: float
    use_scheduler: bool = Field(False)
    scheduler_metric: str | None
    lr_scheduler_args: dict[str, Any] | None
    batch_size: int
    min_epochs: int
    max_epochs: int
    main_metric: str
    early_stopping: dict[str, Any] | None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    num_workers: int
    use_class_weights: bool = False

    @property
    def hparams(self) -> dict[str, Any]:
        """Retrieves all hyperparameters, dropping paths configuration."""
        config = self.dict()
        del config["dataset_dir"]
        del config["experiment_dir"]
        return config

    @property
    def use_wandb(self) -> bool:
        return self.wandb_project is not None

    @classmethod
    def load_from_yaml(cls: Type[T], path: str | Path, config_key: str | None = None) -> T:
        raw_config = load_config(path)
        return cls.load_from_dict(raw_config, config_key)

    @classmethod
    def load_from_dict(
        cls: Type[T], raw_config: dict[str, Any], config_key: str | None = None
    ) -> T:
        if config_key is not None:
            raw_config = raw_config[config_key]
        return cls(**raw_config)
