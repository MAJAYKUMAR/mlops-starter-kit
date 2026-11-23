"""Model definition and training (domain layer, no tracking imports)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


def build_model(n_estimators: int = 100, max_depth: int | None = None):
    """Build an untrained RandomForest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE
    )


def train_model(X_train, y_train, n_estimators: int = 100, max_depth: int | None = None):
    """Train a RandomForest classifier with the given hyperparameters."""
    model = build_model(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model
