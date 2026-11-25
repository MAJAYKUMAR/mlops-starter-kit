"""CLI entry point: score rows with the registered Iris model.

Loads ``models:/iris-random-forest/latest`` from the local MLflow
Model Registry and prints one prediction per row -- either a few
built-in sample rows or rows from a CSV file.

Usage:
    python -m src.cli.predict_cli              # or: make predict
    python -m src.cli.predict_cli --input rows.csv
    python -m src.cli.predict_cli --model models:/iris-random-forest/1

The MLflow import is lazy (inside :func:`load_model`) so the pure
helpers in this module -- sample rows, CSV parsing, argument parsing
-- stay testable without MLflow installed. Run ``make train`` first:
the model must be registered before it can be loaded.
"""

from __future__ import annotations

import argparse
import csv

import pandas as pd

DEFAULT_MODEL_URI = "models:/iris-random-forest/latest"

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# One representative row per Iris class (classic measurements).
SAMPLE_ROWS = [
    {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
    {"sepal length (cm)": 7.0, "sepal width (cm)": 3.2, "petal length (cm)": 4.7, "petal width (cm)": 1.4},
    {"sepal length (cm)": 6.3, "sepal width (cm)": 3.3, "petal length (cm)": 6.0, "petal width (cm)": 2.5},
]


def build_sample_rows() -> list[dict[str, float]]:
    """Return one sample row per Iris class."""
    return [dict(row) for row in SAMPLE_ROWS]


def load_rows_from_csv(path: str) -> list[dict[str, float]]:
    """Read feature rows from a CSV with the Iris feature columns."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in FEATURE_NAMES if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV {path} is missing columns: {', '.join(missing)}")
        rows = []
        for i, raw in enumerate(reader, start=2):
            try:
                rows.append({name: float(raw[name]) for name in FEATURE_NAMES})
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV {path} line {i}: non-numeric value ({exc})") from exc
    if not rows:
        raise ValueError(f"CSV {path} has no data rows")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (``argv`` is injectable for tests)."""
    parser = argparse.ArgumentParser(description="Score rows with the registered Iris model.")
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file with Iris feature rows (default: built-in sample rows)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_URI,
        help=f"Model URI to load (default: {DEFAULT_MODEL_URI})",
    )
    return parser.parse_args(argv)


def load_model(model_uri: str):
    """Load a registered model via MLflow (imported lazily)."""
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(model_uri)


def predict(model, rows: list[dict[str, float]]) -> list[str]:
    """Score ``rows`` with a loaded model; return class names."""
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    predictions = model.predict(frame)
    return [CLASS_NAMES[int(p)] for p in predictions]


def main() -> None:
    args = parse_args()
    rows = load_rows_from_csv(args.input) if args.input else build_sample_rows()
    model = load_model(args.model)
    labels = predict(model, rows)

    for row, label in zip(rows, labels):
        values = ", ".join(f"{row[name]:.1f}" for name in FEATURE_NAMES)
        print(f"[{values}] -> {label}")


if __name__ == "__main__":
    main()
