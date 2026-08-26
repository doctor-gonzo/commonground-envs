"""Common Ground masked-vote prediction environment."""

from importlib.metadata import PackageNotFoundError, version

from commonground_predict.environment import (
    CommonGroundPredictTaskset,
    PredictionJsonParser,
    load_environment,
    render_prompt,
)

try:
    __version__ = version("commonground-predict")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "CommonGroundPredictTaskset",
    "PredictionJsonParser",
    "load_environment",
    "render_prompt",
]
