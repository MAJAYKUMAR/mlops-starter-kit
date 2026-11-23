"""Ports (interfaces) the application layer depends on.

The training service talks to experiment tracking only through these
protocols, so the domain/application code never imports MLflow (or any
other tracking backend) directly. Adapters live in infrastructure.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrackingPort(Protocol):
    """Minimal experiment-tracking + model-registry interface."""

    def set_experiment(self, name: str) -> None:
        """Select (or create) the experiment runs are logged to."""
        ...

    def start_run(self) -> Any:
        """Return a context manager for one tracked run."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the active run."""
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics for the active run."""
        ...

    def log_model(self, model: Any, name: str = "model") -> str:
        """Log a fitted model; return its model URI (e.g. ``runs:/...``)."""
        ...

    def register_model(self, model_uri: str, name: str) -> None:
        """Register a logged model under ``name`` in the model registry."""
        ...
