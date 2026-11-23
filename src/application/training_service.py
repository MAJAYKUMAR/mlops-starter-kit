"""Training service (application layer).

Orchestrates the domain logic (load data, train, evaluate) and the
tracking port (log params/metrics/model, register the model). It knows
nothing about MLflow itself -- any adapter implementing TrackingPort
will do, which keeps the service unit-testable without a tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.application.ports import TrackingPort
from src.domain import evaluate_model, load_data, train_model

EXPERIMENT_NAME = "iris-random-forest"
MODEL_NAME = "iris-random-forest"


@dataclass
class TrainingConfig:
    n_estimators: int = 100
    max_depth: int | None = None
    test_size: float = 0.2
    experiment_name: str = EXPERIMENT_NAME
    model_name: str = MODEL_NAME


class TrainingService:
    """Run one training job end-to-end and track it via ``tracker``."""

    def __init__(self, tracker: TrackingPort, config: TrainingConfig | None = None):
        self.tracker = tracker
        self.config = config or TrainingConfig()

    def run(self) -> dict[str, float]:
        """Train, evaluate, log everything, register the model.

        Returns the evaluation metrics (accuracy and weighted F1).
        """
        cfg = self.config
        self.tracker.set_experiment(cfg.experiment_name)

        X_train, X_test, y_train, y_test = load_data(test_size=cfg.test_size)
        model = train_model(X_train, y_train, cfg.n_estimators, cfg.max_depth)
        metrics = evaluate_model(model, X_test, y_test)

        with self.tracker.start_run():
            self.tracker.log_params(
                {
                    "n_estimators": cfg.n_estimators,
                    "max_depth": cfg.max_depth,
                    "test_size": cfg.test_size,
                }
            )
            self.tracker.log_metrics(metrics)
            model_uri = self.tracker.log_model(model, name="model")
            self.tracker.register_model(model_uri, cfg.model_name)

        return metrics
