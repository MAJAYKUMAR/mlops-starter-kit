"""MLflow tracking + model-registry adapter.

The only module in the codebase that imports MLflow. It implements
the application layer's TrackingPort, so swapping tracking backends
means writing a new adapter here -- nothing else changes.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn

# MLflow 3.x + scikit-learn >= 1.6 serialize tree models via skops,
# which requires explicitly trusted types for the tree internals.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


class MlflowTracker:
    """TrackingPort adapter backed by MLflow tracking and the registry."""

    def set_experiment(self, name: str) -> None:
        mlflow.set_experiment(name)

    def start_run(self) -> Any:
        return mlflow.start_run()

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, name: str = "model") -> str:
        mlflow.sklearn.log_model(
            model, name=name, skops_trusted_types=SKOPS_TRUSTED_TYPES
        )
        run_id = mlflow.active_run().info.run_id
        return f"runs:/{run_id}/{name}"

    def log_artifact(self, name: str, content: str) -> None:
        mlflow.log_text(content, artifact_file=name)

    def register_model(self, model_uri: str, name: str) -> None:
        mlflow.register_model(model_uri, name)
