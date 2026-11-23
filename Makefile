.PHONY: train mlflow-ui lint test install

install:
	pip install -r requirements.txt

train:
	python -m src.cli.train_cli

mlflow-ui:
	mlflow ui --host 127.0.0.1 --port 5000

lint:
	ruff check src tests

test:
	pytest -q
