"""Evaluation metrics (domain layer, no tracking imports)."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a fitted model.

    Returns accuracy and weighted F1 plus the per-class
    classification report (keyed by Iris class name) and the
    confusion matrix as a list of rows. Aggregate numbers alone hide
    per-class failures, so the report and matrix are part of the
    result rather than an afterthought.
    """
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "classification_report": classification_report(
            y_test, preds, target_names=CLASS_NAMES, output_dict=True
        ),
        "classification_report_text": classification_report(
            y_test, preds, target_names=CLASS_NAMES
        ),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def confusion_matrix_to_csv(cm) -> str:
    """Render a confusion matrix as CSV with a header row of class names."""
    lines = ["actual\\predicted," + ",".join(CLASS_NAMES)]
    for name, row in zip(CLASS_NAMES, cm):
        lines.append(name + "," + ",".join(str(int(v)) for v in row))
    return "\n".join(lines) + "\n"
