"""Training service (application layer).

Orchestrates the domain logic (load data, train, evaluate) and the
tracking port (log params/metrics/model, register the model). It knows
nothing about MLflow itself -- any adapter implementing TrackingPort
will do, which keeps the service unit-testable without a tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.application.ports import TrackingPort
from src.domain import (
    confusion_matrix_to_csv,
    evaluate_model,
    load_data,
    train_model,
)

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

    def run(self) -> dict:
        """Train, evaluate, log everything, register the model.

        Returns the full evaluation result: accuracy, weighted F1,
        the per-class classification report, and the confusion matrix.
        Only the two scalar metrics go to ``log_metrics``; the report
        and matrix are logged as text artifacts so they stay readable
        in the tracking UI.
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
            self.tracker.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "f1_weighted": metrics["f1_weighted"],
                }
            )
            self.tracker.log_artifact(
                "classification_report.txt", metrics["classification_report_text"]
            )
            self.tracker.log_artifact(
                "confusion_matrix.csv",
                confusion_matrix_to_csv(metrics["confusion_matrix"]),
            )
            model_uri = self.tracker.log_model(model, name="model")
            self.tracker.register_model(model_uri, cfg.model_name)

        return metrics
