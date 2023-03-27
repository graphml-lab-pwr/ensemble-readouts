import dataclasses
import json
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from tbparse import SummaryReader
from tqdm import tqdm

from readouts.utils.utils import load_config


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentResults:
    all: dict[str, str | float | None]
    hparams_keys: list[str]
    param_count_modules: list[str]

    @property
    def id_cols(self) -> list[str]:
        return self.hparams_keys + self.param_count_modules


def main(
    experiment_root_dir: Path = typer.Option(..., help="Root directory of all experiments"),
    n_jobs: int = typer.Option(1),
):
    """Gathers metrics from multiple, also repeated, experiments into single table."""
    results, id_cols = summarize_experiments(experiment_root_dir, n_jobs)
    results.to_csv(experiment_root_dir / "all_metrics.csv", index=False)

    numeric_cols = results.select_dtypes(include=np.number).columns.tolist()
    results_agg = results.groupby(id_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    results_agg = results_agg.round(4)
    results_agg.columns = ["_".join(a).strip("_") for a in results_agg.columns.to_flat_index()]
    results_agg.to_csv(experiment_root_dir / "all_metrics_agg.csv", index=False)

    # reorder columns for better quick lookup
    numeric_cols = results_agg.select_dtypes(include=np.number).columns.tolist()
    cols_order = ["experiment_name", "dataset_name"] + numeric_cols
    cols_order += list(set(results_agg.columns).difference(set(cols_order)))
    results_agg[cols_order].sort_values(["dataset_name", "experiment_name"]).to_markdown(
        experiment_root_dir / "all_metrics_agg.md", index=False
    )


def summarize_experiments(experiment_root_dir: Path, n_jobs: int) -> tuple[pd.DataFrame, list[str]]:
    """Extract all metrics file under given dir and assemble into single dataframe."""
    results: list[dict[str, Any]] = []

    loader_func = partial(load_experiment_results, experiment_root_dir=experiment_root_dir)
    experiments = list(experiment_root_dir.glob("**/metrics.json"))
    experiment_res: ExperimentResults | None = None
    with multiprocessing.Pool(n_jobs) as pool:
        iter_exps = pool.imap_unordered(loader_func, experiments)
        for experiment_res in tqdm(iter_exps, desc="Loading results...", total=len(experiments)):
            assert experiment_res is not None
            results.append(experiment_res.all)

    if experiment_res is None:
        raise FileNotFoundError("Not found any experiment results")

    return pd.DataFrame(results), experiment_res.id_cols


def load_experiment_results(metrics_path: Path, experiment_root_dir: Path) -> ExperimentResults:
    # sanity check
    *_, log_dir, _, metrics_file = metrics_path.relative_to(experiment_root_dir).parts
    assert log_dir == "lightning_logs"
    assert metrics_file == "metrics.json"

    experiment_res: dict[str, str | float | None] = {}

    hparams_path = metrics_path.with_name("hparams.yaml")
    hparams = parse_hparams(hparams_path)
    experiment_res |= hparams

    metrics, parameters_count_modules = parse_metrics(metrics_path)
    experiment_res |= metrics
    experiment_res["duration"] = get_duration(metrics_path.parent)

    return ExperimentResults(
        all=experiment_res,
        hparams_keys=list(hparams.keys()),
        param_count_modules=parameters_count_modules,
    )


def parse_metrics(path: Path) -> tuple[dict[str, float], list[str]]:
    with path.open() as file:
        metric_data = json.load(file)
        metrics, *_ = metric_data["metrics"]

        # add prefix to avoid duplication with other names
        parameters_count: dict[str, int] = {
            f"params_{module}": count
            for module, count in metric_data["metadata"]["parameters_count"].items()
        }
        parameters_count_modules = list(parameters_count.keys())
        metrics |= parameters_count

    return metrics, parameters_count_modules


def parse_hparams(path: Path) -> dict[str, str | float | None]:
    hparams = load_config(path)
    return {
        "experiment_name": hparams["experiment_name"],
        "dataset_type": hparams["dataset_type"],
        "dataset_name": hparams["dataset_name"],
        "model_name": hparams["model_name"],
        "graph_conv": hparams["graph_conv"],
        "hidden_dim": str(hparams["hidden_dim"]),  # use string to exclude from aggregations
        "num_layers": str(hparams["num_layers"]),
        "proj_dim": str(hparams["proj_dim"]),
        "readout_name": hparams.get("readout_name", "n/a"),
        "predictor_name": hparams["predictor_name"],
        "main_metric": hparams["main_metric"],
    }


def get_duration(tb_logs_path) -> float:
    """Returns experiment duration from TensorBoard, takes very long time!!!."""
    summary = SummaryReader(tb_logs_path, extra_columns={"wall_time"})
    return summary.scalars["wall_time"].max() - summary.scalars["wall_time"].min()


if __name__ == "__main__":
    typer.run(main)
