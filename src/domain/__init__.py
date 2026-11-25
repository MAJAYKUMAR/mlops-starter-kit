"""Domain layer: pure model/pipeline logic.

No MLflow imports here -- this package only knows about data, models,
and metrics. Anything that talks to the outside world (tracking,
registry, CLI) lives in the application, infrastructure, and cli layers.
"""

from src.domain.dataset import load_data
from src.domain.evaluation import (
    CLASS_NAMES,
    confusion_matrix_to_csv,
    evaluate_model,
)
from src.domain.model import build_model, train_model

__all__ = [
    "CLASS_NAMES",
    "build_model",
    "confusion_matrix_to_csv",
    "evaluate_model",
    "load_data",
    "train_model",
]
