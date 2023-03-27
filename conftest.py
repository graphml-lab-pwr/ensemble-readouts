from pathlib import Path

from pytest import Metafunc

from readouts.utils.utils import load_config

PROJECT_FILE = Path(__file__).parent.resolve()
CONFIG_ROOT = PROJECT_FILE / "experiments" / "config"
CONFIG_FILE_ROOT = CONFIG_ROOT / "ensemble_readouts"
CONFIG_FILE_NAMES = [
    "hparams_enzymes.yaml",
    "hparams_mutag.yaml",
    "hparams_reddit_multi.yaml",
    "hparams_zinc.yaml",

]
CONFIG_EXP_NAMES = CONFIG_ROOT / "config_names.yaml"
EXP_NAME_GROUPS = [
    "ensemble_readouts",
]


def pytest_generate_tests(metafunc: Metafunc):
    # generates all model configurations
    config_exp_names = load_config(CONFIG_EXP_NAMES)
    configs: list[tuple[Path, str]] = []
    for exp_name_grp in EXP_NAME_GROUPS:
        for config_file_name in CONFIG_FILE_NAMES:
            config_path = CONFIG_FILE_ROOT / config_file_name
            for exp_name in config_exp_names[exp_name_grp]:
                configs.append((config_path, exp_name))
    metafunc.parametrize(["config_path", "config_key"], configs)
