"""Evaluation metrics (domain layer, no tracking imports)."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Return accuracy and weighted F1 for a fitted model."""
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
    }
