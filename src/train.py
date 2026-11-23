"""Backwards-compatible training entry point.

The training logic now lives in the layered packages
(``src.domain``, ``src.application``, ``src.infrastructure``) and the
CLI lives in ``src.cli.train_cli``. This module keeps the original
``python -m src.train`` command working: it re-exports the domain
helpers and delegates ``main()`` to the CLI.

Usage:
    python -m src.train            # or: make train
    python -m src.train --n-estimators 100 --max-depth 5
"""

from __future__ import annotations

from src.application import EXPERIMENT_NAME, MODEL_NAME
from src.cli.train_cli import main
from src.domain import evaluate_model, load_data, train_model

__all__ = ["EXPERIMENT_NAME", "MODEL_NAME", "evaluate_model", "load_data", "main", "train_model"]


if __name__ == "__main__":
    main()
