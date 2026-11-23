"""CLI entry point: train an Iris RandomForest with MLflow tracking.

Wires the layers together -- domain logic via the application training
service, MLflow via the infrastructure adapter -- and runs one
training job end to end.

Usage:
    python -m src.cli.train_cli          # or: make train
    python -m src.cli.train_cli --n-estimators 100 --max-depth 5
    python -m src.train                  # backwards-compatible alias
"""

from __future__ import annotations

import argparse

from src.application import TrainingConfig, TrainingService
from src.infrastructure import MlflowTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Iris RandomForest with MLflow tracking.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    config = TrainingConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
    )
    metrics = TrainingService(MlflowTracker(), config).run()

    print(f"Accuracy: {metrics['accuracy']:.4f} | F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Model registered as '{config.model_name}'. Run `make mlflow-ui` to browse runs.")


if __name__ == "__main__":
    main()
