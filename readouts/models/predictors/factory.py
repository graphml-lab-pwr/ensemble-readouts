from typing import Any

from readouts.models.predictors.predictor_base import PredictorBase
from readouts.utils.utils import import_from_string


def create_predictor(predictor_name: str, **predictor_init_args: dict[str, Any]) -> PredictorBase:
    """Factory method for building predictor from its name and config."""
    predictor_cls = import_from_string(f"readouts.models.predictors.{predictor_name}")

    assert issubclass(predictor_cls, PredictorBase)
    return predictor_cls(**predictor_init_args)
