"""Tests for the predict CLI.

The CLI's pure helpers (sample rows, CSV parsing, argument parsing)
and prediction against a fake model are covered here; none of these
tests need MLflow installed. Loading a real model from the registry
is exercised end to end by ``make train`` + ``make predict``.
"""

import numpy as np

from src.cli.predict_cli import (
    CLASS_NAMES,
    DEFAULT_MODEL_URI,
    FEATURE_NAMES,
    build_sample_rows,
    load_rows_from_csv,
    parse_args,
    predict,
)


def test_build_sample_rows_has_one_row_per_class():
    rows = build_sample_rows()
    assert len(rows) == 3
    for row in rows:
        assert list(row) == FEATURE_NAMES
        assert all(isinstance(v, float) for v in row.values())


def test_parse_args_defaults():
    args = parse_args([])
    assert args.input is None
    assert args.model == DEFAULT_MODEL_URI


def test_parse_args_input_and_model():
    args = parse_args(["--input", "rows.csv", "--model", "models:/iris-random-forest/1"])
    assert args.input == "rows.csv"
    assert args.model == "models:/iris-random-forest/1"


def test_load_rows_from_csv(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n"
        "5.1,3.5,1.4,0.2\n"
        "6.3,3.3,6.0,2.5\n"
    )
    rows = load_rows_from_csv(str(csv_path))
    assert rows == [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.3, "sepal width (cm)": 3.3, "petal length (cm)": 6.0, "petal width (cm)": 2.5},
    ]


def test_load_rows_from_csv_rejects_missing_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("sepal length (cm),sepal width (cm)\n5.1,3.5\n")
    try:
        load_rows_from_csv(str(csv_path))
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing columns")


def test_load_rows_from_csv_rejects_non_numeric(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n"
        "five,3.5,1.4,0.2\n"
    )
    try:
        load_rows_from_csv(str(csv_path))
    except ValueError as exc:
        assert "non-numeric" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-numeric value")


class FakeModel:
    """Stand-in for a loaded pyfunc model (no MLflow needed)."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, frame):
        assert list(frame.columns) == FEATURE_NAMES
        assert len(frame) == len(self._predictions)
        return np.array(self._predictions)


def test_predict_maps_predictions_to_class_names():
    rows = build_sample_rows()
    labels = predict(FakeModel([0, 1, 2]), rows)
    assert labels == CLASS_NAMES
