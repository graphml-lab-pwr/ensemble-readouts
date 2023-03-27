from pathlib import Path

from readouts.models import GraphModelConfig, ModelBase, get_model_cls
from readouts.utils.utils import load_config


def test_load_config_and_build_models(config_path: Path, config_key: str):
    """Checks whether specified configuration loads properly."""
    raw_config = load_config(config_path, config_key, "experiment_name")
    model_cls = get_model_cls(raw_config["model_name"])

    # get first random seed
    raw_config["random_seed"], *_ = raw_config["random_seed"]
    config: GraphModelConfig = model_cls.config_cls(**raw_config)
    model = model_cls(config)

    # dummy check to test its type
    assert isinstance(model, ModelBase)
