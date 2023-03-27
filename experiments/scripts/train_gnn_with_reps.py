"""
Experiments (with configurable repeats) on graph data
"""

import argparse
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import torch.cuda
from pytorch_lightning import Trainer
from tqdm import tqdm

from experiments.gnn_training import train_gnn
from readouts.utils.utils import load_config


def main(experiment_args: Namespace):
    experiment_args = set_device_config(experiment_args)
    raw_config = load_config(
        experiment_args.config_path, experiment_args.config_key, "experiment_name"
    )

    assert raw_config["experiment_dir"].endswith(raw_config["experiment_name"])
    assert isinstance(raw_config["random_seed"], list)
    assert len(raw_config["random_seed"]) >= raw_config["repeats"]
    for i, random_seed in enumerate(
        tqdm(raw_config["random_seed"][: raw_config["repeats"]], desc="Experiment repeat")
    ):
        experiment_raw_config = deepcopy(raw_config)
        experiment_raw_config["random_seed"] = random_seed

        try:
            train_gnn(experiment_args, experiment_raw_config)
        except KeyboardInterrupt:
            sys.exit(
                "Training was interrupted, "
                f"aborting experiment at repeat {i}/{raw_config['repeats']}"
            )


def set_device_config(experiment_args: Namespace) -> Namespace:
    if experiment_args.accelerator is None:
        if torch.cuda.is_available():
            experiment_args.accelerator = "gpu"
            experiment_args.devices = 1
        else:
            experiment_args.accelerator = "cpu"
            experiment_args.devices = None
    return experiment_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run chain classification training")
    parser.add_argument(
        "--config-path",
        type=Path,
        nargs="?",
        required=True,
        help="Path to YAML with experiment config",
    )
    parser.add_argument(
        "--config-key",
        type=str,
        required=False,
        default="default",
        help="Name of the config key/section in the experiment config YAML file",
    )
    parser_extended = Trainer.add_argparse_args(parser)
    assert isinstance(parser_extended, argparse.ArgumentParser)
    args = parser_extended.parse_args()
    main(args)
