"""Tests for the layered training code.

Domain helpers live in ``src.domain``; the training service in
``src.application`` is tested with a fake tracker so no MLflow run
is needed; ``src.train`` keeps re-exporting the old entry points.
"""

from contextlib import contextmanager

import numpy as np

from src.application import TrainingConfig, TrainingService
from src.domain import evaluate_model, load_data, train_model


def test_load_data_shape():
    X_train, X_test, y_train, y_test = load_data()
    # Iris has 150 rows and 4 features; 80/20 split -> 120 train, 30 test.
    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_load_data_is_deterministic():
    first = load_data()
    second = load_data()
    assert np.array_equal(first[0].to_numpy(), second[0].to_numpy())


def test_model_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators=10)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_evaluate_model_scores_are_reasonable():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators=10)
    metrics = evaluate_model(model, X_test, y_test)
    # A RandomForest on Iris should do well on a held-out split.
    assert 0.8 <= metrics["accuracy"] <= 1.0
    assert 0.8 <= metrics["f1_weighted"] <= 1.0


class FakeTracker:
    """In-memory stand-in for the TrackingPort (no MLflow needed)."""

    def __init__(self):
        self.experiment = None
        self.params = {}
        self.metrics = {}
        self.logged_model_uri = None
        self.registered = []

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self):
        yield

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_model(self, model, name="model"):
        self.logged_model_uri = f"runs:/fake-run/{name}"
        return self.logged_model_uri

    def register_model(self, model_uri, name):
        self.registered.append((model_uri, name))


def test_training_service_logs_and_registers_without_mlflow():
    tracker = FakeTracker()
    service = TrainingService(tracker, TrainingConfig(n_estimators=10))
    metrics = service.run()

    assert tracker.experiment == "iris-random-forest"
    assert tracker.params["n_estimators"] == 10
    assert 0.8 <= tracker.metrics["accuracy"] <= 1.0
    assert metrics == tracker.metrics
    assert tracker.registered == [("runs:/fake-run/model", "iris-random-forest")]


def test_train_module_reexports_domain_helpers():
    # `python -m src.train` stays a working entry point.
    import src.train as train_module

    assert train_module.load_data is load_data
    assert train_module.train_model is train_model
    assert train_module.evaluate_model is evaluate_model
    assert train_module.MODEL_NAME == "iris-random-forest"
    assert callable(train_module.main)
