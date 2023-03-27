from typing import Any, TypeVar

import torch
import torch.nn as nn
from pydantic import Field
from pytorch_lightning.utilities.model_summary import summarize
from torch_geometric.data import Batch

from readouts.models.model_base import GNNConfig, ModelBase
from readouts.models.predictors import PredictorBase, create_predictor
from readouts.models.readout_layers import ReadoutLayer, create_readout
from readouts.models.readout_layers.output import ReadoutOutput

T = TypeVar("T", bound="GraphModelConfig")


class GraphModelConfig(GNNConfig, frozen=True):
    """Configuration for graph-level experiments."""

    readout_name: str
    readout_init_args: dict[str, Any] = Field(default_factory=dict)
    readout_dim: int
    predictor_name: str
    predictor_init_args: dict[str, Any] = Field(default_factory=dict)


class GraphModel(ModelBase[GraphModelConfig]):
    """Model and training of the graph-level tasks which are based on trained node embeddings."""

    config_cls = GraphModelConfig

    def __init__(self, config: GraphModelConfig | dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self.readout: nn.Module = self.init_readout()

    @property
    def param_stats(self) -> dict[str, int]:
        model_summary = summarize(self)
        params = dict(zip(model_summary.layer_names, model_summary.param_nums))
        params_summary = {
            "graph_conv": params["graph_conv"],
            "readout": params["readout"],
            "predictor": params["predictor"],
        }
        assert sum(params_summary.values()) == model_summary.trainable_parameters
        return params_summary

    def init_readout(self) -> ReadoutLayer:
        return create_readout(self.config.readout_name, **self.config.readout_init_args)

    def init_predictor(self) -> PredictorBase:
        return create_predictor(self.config.predictor_name, **self.config.predictor_init_args)

    def forward(self, graph_batch: Batch) -> tuple[torch.Tensor, ReadoutOutput]:
        readout_output = self.forward_repr(graph_batch)
        graph_preds = self.predictor(readout_output.graph_reprs)
        return graph_preds, readout_output

    def forward_repr(self, graph_batch: Batch) -> ReadoutOutput:
        graph_node_reprs = self.graph_conv(
            graph_batch.x, graph_batch.edge_index, batch=graph_batch.batch
        )
        assert isinstance(graph_node_reprs, torch.Tensor)
        readout_output = self.readout(graph_node_reprs, graph_batch)
        assert isinstance(readout_output, ReadoutOutput)
        return readout_output

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        out, loss = self._shared_step(batch)
        self.log("train_Loss", loss, batch_size=len(batch))
        self.train_metrics(out.detach(), batch.y)
        self.log_dict(self.train_metrics, batch_size=len(batch))  # type: ignore[arg-type]

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        out, loss = self._shared_step(batch)
        self.log("val_Loss", loss, batch_size=len(batch), prog_bar=True)
        self.val_metrics(out.detach(), batch.y)
        self.log_dict(self.val_metrics, batch_size=len(batch))  # type: ignore[arg-type]

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        out, readout_output = self.forward(batch)
        self.test_metrics(out.detach(), batch.y)
        self.log_dict(self.test_metrics, batch_size=len(batch))  # type: ignore[arg-type]

    def _shared_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        preds, readout_output = self.forward(batch)

        if self.config.output_dim == 1:
            target = batch.y.float()  # floats are required by BCEWithLogitsLoss
        else:
            target = batch.y

        loss = self.predictor.loss(preds, target)

        return preds, loss
