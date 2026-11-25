"""CLI entry points.

Kept import-free on purpose: importing ``src.cli`` must not pull in
MLflow (via ``train_cli``), so the predict CLI's helpers stay
importable and testable without MLflow installed. Import the entry
points directly, e.g. ``from src.cli.train_cli import main``.
"""
