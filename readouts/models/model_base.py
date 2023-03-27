import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Generic, Literal, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn
import wandb
from pydantic import Field
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from wandb.sdk.wandb_run import Run

from readouts.models.config import TrainingConfig
from readouts.models.gnn import GraphNeuralNetwork
from readouts.models.predictors import PredictorBase


class GNNConfig(TrainingConfig, ABC, frozen=True):
    graph_conv: Literal["gcn", "gin", "gat"]
    hidden_dim: int
    num_layers: int
    proj_dim: int | None
    repr_dim: int
    conv_kwargs: dict[str, Any] = Field(default_factory=dict)
    transforms_config: dict[str, Any] | None = Field(default=None)


T_config = TypeVar("T_config", bound=GNNConfig)


class ModelBase(LightningModule, ABC, Generic[T_config]):
    """Base class for all models relying on LightningModule"""

    config_cls: Type[T_config]

    def __init__(self, config: T_config | dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert self.config_cls is not None

        self.config = self.parse_config(config)
        self.save_hyperparameters({"config": self.config.dict()})

        self.graph_conv: nn.Module = self.init_graph_conv()
        self.predictor: PredictorBase = self.init_predictor()

        # split metrics to train/val/test
        self.train_metrics = self.predictor.metrics.clone(prefix="train_")
        self.val_metrics = self.predictor.metrics.clone(prefix="val_")
        self.test_metrics = self.predictor.metrics.clone(prefix="test_")

        # placeholder for histogram logging
        self.histograms: dict[str, list[int | float]] = defaultdict(list)

    @property
    @abstractmethod
    def param_stats(self) -> dict[str, int]:
        """Provides number of parameters for a model, distinguishing layers"""

    def init_graph_conv(self) -> nn.Module:
        # For molhiv dataset use custom atom and bond encoders
        return GraphNeuralNetwork(
            self.config.graph_conv,
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.proj_dim,
            self.config.num_layers,
            prediction_heads=1,
            heads_dim=None,
            **self.config.conv_kwargs,
        )

    @abstractmethod
    def init_predictor(self) -> PredictorBase:
        raise NotImplementedError

    def configure_optimizers(self) -> dict[str, Any]:
        optim = AdamW(params=self.parameters(), lr=self.config.learning_rate)

        optimizer: dict[str, Any] = {"optimizer": optim}

        if self.config.use_scheduler:
            assert isinstance(self.config.lr_scheduler_args, dict)
            optimizer["lr_scheduler"] = {
                "scheduler": ReduceLROnPlateau(optim, **self.config.lr_scheduler_args),
                "interval": "epoch",
                "frequency": 1,
            }
            if self.config.scheduler_metric:
                optimizer["lr_scheduler"]["monitor"] = self.config.scheduler_metric

        return optimizer

    def parse_config(self, config: T_config | dict[str, Any]) -> T_config:
        if isinstance(config, self.config_cls):
            return config
        elif isinstance(config, dict):
            return self.config_cls(**config)
        else:
            raise TypeError(f"Invalid configuration type: {type(config)}")

    def log_data_to_histogram(self, data: torch.Tensor | list[float], name: str) -> None:
        """Extend current histogram state (recognized by name) with given data."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            hist_data = data.tolist()
        else:
            hist_data = data

        self.histograms[name].extend(hist_data)

    def training_epoch_end(self, outputs: list[torch.Tensor | dict[str, Any]]) -> None:
        """Logs histograms and clear histogram memory for the next epoch."""
        if len(self.histograms):
            self.log_histograms(self.histograms, "tensorboard")
            self.log_histograms(self.histograms, "wandb")
            self.histograms.clear()

    def log_histograms(
        self,
        histograms_data: dict[str, list[int | float]],
        logger_type: Literal["tensorboard", "wandb"],
    ) -> None:
        if logger_type == "tensorboard":
            logger = self.get_tb_writer()
        elif logger_type == "wandb":
            logger = self.get_wandb_run()
        else:
            raise ValueError(f"Invalid logger_type to log histogram: {logger_type}")

        if logger is None:
            logging.warning(f"{logger_type} not available, cannot log histograms")
            return

        for hist_name, hist_values in histograms_data.items():
            if isinstance(logger, SummaryWriter):
                logger.add_histogram(hist_name, np.asarray(hist_values), self.current_epoch)
            elif isinstance(logger, Run):
                np_hist = np.histogram(hist_values)
                # when using 'step' arg logging doesn't work, put epoch in dict instead
                logger.log(
                    {
                        f"histogram/{hist_name}": wandb.Histogram(np_histogram=np_hist),
                        "epoch": self.current_epoch,
                        "trainer/global_step": self.global_step,
                    },
                    commit=False,
                )
            else:
                raise TypeError(
                    f"None of handled logger type matched retrieved logger for type {logger_type}"
                )

    def get_tb_writer(self) -> SummaryWriter | None:
        writer = self._get_backbone_logger(TensorBoardLogger)
        if writer is None:
            return None
        assert isinstance(writer, SummaryWriter)
        return writer

    def get_wandb_run(self) -> Run | None:
        run = self._get_backbone_logger(WandbLogger)
        if run is None:
            return None
        assert isinstance(run, Run)
        return run

    def _get_backbone_logger(self, pl_logger_cls: Type[Logger]) -> SummaryWriter | Run | None:
        loggers = self.loggers
        if isinstance(loggers, list):
            for lg in loggers:
                if isinstance(lg, pl_logger_cls):
                    assert hasattr(lg, "experiment")
                    return lg.experiment
            return None
        elif isinstance(loggers, pl_logger_cls):
            assert hasattr(loggers, "experiment")
            return loggers.experiment
        else:
            raise TypeError(f"Cannot find {pl_logger_cls} logger")
