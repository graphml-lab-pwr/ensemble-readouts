import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import wandb
import wandb.util
from lightning_lite import seed_everything
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
from torch_geometric.data import LightningDataset

from readouts.datasets.benchmarks import get_benchmark_datamodule
from readouts.models import GraphModelConfig, ModelBase, get_model_cls
from readouts.models.config import TrainingConfig
from readouts.utils.utils import get_train_labels_loss_weights

# in case of unstable connection use the WANDB_MODE=offline env variable
# in case of debugging, one should change WANDB project or use offline and don't sync
WANDB_DEBUG_PROJECT = os.getenv("WANDB_DEBUG_PROJECT", None)
# in case of debugging decrease workers to 0 (avoid PyCharm hangs)
DEBUG_MODE = bool(os.getenv("DEBUG_MODE", False))


def train_gnn(experiment_args: Namespace, raw_config: dict[str, Any]) -> None:
    """Trains GNN-based model given the configuration."""
    seed_everything(raw_config["random_seed"], workers=True)

    raw_config = alter_config_when_debug(raw_config)

    model_cls = get_model_cls(raw_config["model_name"])
    config: GraphModelConfig = model_cls.config_cls(**raw_config)

    datamodule = get_benchmark_datamodule(
        root_dir=config.dataset_dir,
        task_level=config.task_level,
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        output_dim=config.output_dim,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transforms_config=config.transforms_config,
        **config.dataset_kwargs,
    )

    if config.use_class_weights:
        config.predictor_init_args["class_weights"] = get_train_labels_loss_weights(
            datamodule, config.output_dim
        )

    model = model_cls(config)

    callbacks = setup_callbacks(config)
    loggers = setup_loggers(config, model)

    trainer: Trainer = Trainer.from_argparse_args(
        experiment_args,
        default_root_dir=config.experiment_dir,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        logger=loggers,
        callbacks=callbacks,
        deterministic="warn",
    )
    trainer.fit(model, datamodule=datamodule)

    was_interrupted = trainer.interrupted

    evaluate_model(config, trainer, datamodule)

    if config.use_wandb:
        wandb.finish()

    # after graceful experiment abortion, re-raise interrupt
    if was_interrupted:
        raise KeyboardInterrupt()


def alter_config_when_debug(raw_config: dict[str, Any]) -> dict[str, Any]:
    if DEBUG_MODE:
        raw_config["num_workers"] = 0

        if raw_config.get("wandb_project"):
            assert (
                WANDB_DEBUG_PROJECT is not None
                and WANDB_DEBUG_PROJECT != raw_config["wandb_project"]
            ), "When running in debug mode, ensure wandb project other than in config"
            raw_config["wandb_project"] = WANDB_DEBUG_PROJECT
    return raw_config


def setup_callbacks(config: TrainingConfig) -> list[Callback]:
    callbacks: list[Callback] = [LearningRateMonitor(logging_interval="epoch")]
    if config.early_stopping:
        callbacks.append(EarlyStopping(**config.early_stopping))

    if config.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            filename=f"{{epoch}}-{{{config.checkpoint['monitor']}:.2f}}",
            **config.checkpoint,
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
    return callbacks


def setup_loggers(config: TrainingConfig, model: ModelBase) -> list[Logger]:
    # need use random string, since wandb won't accept same id in the future, also after delete
    version = wandb.util.generate_id()
    tb_logger = TensorBoardLogger(
        str(config.experiment_dir), default_hp_metric=False, version=version
    )
    tb_logger.log_hyperparams(config.hparams)
    loggers: list[Logger] = [tb_logger]

    if config.use_wandb:
        assert config.wandb_project is not None
        config.experiment_dir.mkdir(parents=True, exist_ok=True)
        wandb_logger = WandbLogger(
            name=config.experiment_name,
            group=config.experiment_name,
            version=version,
            save_dir=config.experiment_dir,
            project=config.wandb_project,
            log_model=True,
            reinit=True,
        )
        # enable logging gradients for each layer
        wandb_logger.watch(model)
        loggers.append(wandb_logger)
    return loggers


def evaluate_model(config: TrainingConfig, trainer: Trainer, datamodule: LightningDataset) -> None:
    metrics: dict[str, list[dict[str, Any]] | dict[str, Any]] = {}
    model: ModelBase = trainer.lightning_module
    assert isinstance(model, ModelBase)
    # cast to int in case of int64 dtype which comes from torch.long and is not json serializable
    parameters_count = {module: int(count) for module, count in model.param_stats.items()}
    metrics["metadata"] = {"main_metric": config.main_metric, "parameters_count": parameters_count}
    test_metrics = trainer.test(
        model, datamodule=datamodule, ckpt_path=ModelCheckpoint.CHECKPOINT_NAME_LAST
    )
    metrics["metrics"] = test_metrics

    assert trainer.log_dir is not None
    log_dir = Path(trainer.log_dir)
    with open(log_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent="\t")
