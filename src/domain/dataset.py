"""Dataset loading (domain layer, no tracking imports)."""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data(test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Load the Iris dataset and split it into train/test sets."""
    data = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
