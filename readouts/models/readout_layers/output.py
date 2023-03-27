from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutOutput:
    graph_reprs: torch.Tensor
