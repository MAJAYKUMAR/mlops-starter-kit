# MLOps Starter Kit

A minimal, clean-room MLOps starter template: train a model with
[MLflow](https://mlflow.org/) tracking and the Model Registry, test it
with pytest, lint it with ruff, and deploy it with a Helm chart and a
Terraform skeleton for EKS-style infrastructure.

Everything here is written from scratch as a learning template. It uses
only public example data (the Iris dataset) and placeholder values — no
real infrastructure, credentials, or production details.

## Use this template

Click **Use this template → Create a new repository** on GitHub to start
your own project from this repo, or clone it directly:

```bash
git clone https://github.com/MAJAYKUMAR/mlops-starter-kit.git
cd mlops-starter-kit
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make train        # trains a RandomForest on Iris, logs to MLflow, registers the model
make mlflow-ui    # browse runs at http://127.0.0.1:5000
```

Training uses a local `mlruns/` directory — no MLflow server required.
The model is registered in the local MLflow Model Registry under the
name `iris-random-forest`.

## Repo layout

```
.
├── src/
│   ├── domain/             # pure logic: dataset, model, evaluation (no MLflow)
│   │   ├── dataset.py      # load_data: Iris dataset + train/test split
│   │   ├── model.py        # build_model / train_model: RandomForest
│   │   └── evaluation.py   # evaluate_model: accuracy + weighted F1
│   ├── application/        # training service + ports (orchestration, no MLflow)
│   │   ├── ports.py        # TrackingPort: tracking/registry interface
│   │   └── training_service.py  # TrainingService: load, train, evaluate, track
│   ├── infrastructure/     # adapters: every MLflow call lives here
│   │   └── mlflow_tracker.py    # MlflowTracker: TrackingPort via MLflow
│   ├── cli/                # entry points
│   │   └── train_cli.py    # `python -m src.cli.train_cli` (also: make train)
│   └── train.py            # backwards-compatible alias for `python -m src.train`
├── tests/
│   └── test_train.py       # pytest tests: domain helpers + training service
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI: lint + tests on push/PR
├── charts/ml-service/      # Helm chart skeleton for a model-serving container
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/          # deployment.yaml, service.yaml
├── terraform/              # Terraform skeleton for EKS-style infra (example only)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── Makefile                # make train / make mlflow-ui / make lint / make test
├── pyproject.toml          # pytest config (project root on sys.path)
├── requirements.txt        # runtime deps: scikit-learn, mlflow, pandas
└── requirements-dev.txt    # dev deps: pytest, ruff
```

## Architecture

The code is split into layers so the model logic can be tested and
reused without MLflow, and the tracking backend can be swapped by
writing one new adapter:

- **Domain** (`src/domain/`) — pure functions for loading data,
  building/training the model, and computing metrics. No MLflow
  imports, no I/O beyond the dataset itself.
- **Application** (`src/application/`) — `TrainingService` orchestrates
  the domain steps and talks to tracking only through the
  `TrackingPort` protocol. The service is unit-testable with a fake
  tracker (see `tests/test_train.py`).
- **Infrastructure** (`src/infrastructure/`) — `MlflowTracker` is the
  only module that imports MLflow. It implements `TrackingPort`:
  experiment selection, run logging, model logging, and registry.
- **CLI** (`src/cli/`) — thin entry points that wire the layers
  together and parse arguments. `src/train.py` remains as a
  backwards-compatible alias, so `python -m src.train` still works.

## CI

`.github/workflows/ci.yml` runs on every push and pull request: it
installs Python 3.12, installs the requirements, runs `ruff check`
on `src` and `tests`, and runs `pytest`.

Run the same checks locally:

```bash
make lint
make test
```

## Helm

The chart in `charts/ml-service/` deploys a generic model-serving
container. Point `image.repository` / `image.tag` at your own image
in `values.yaml`, then:

```bash
helm template my-release charts/ml-service/
helm install my-release charts/ml-service/
```

## Terraform

The files in `terraform/` are a **skeleton / example only** — a VPC +
EKS cluster starting point with variables for region, cluster name,
Kubernetes version, and node sizing. It is not production-ready and
has not been applied. Review and harden it (private subnets, IRSA,
logging, state backend) before using it for real infrastructure.

The skeleton has been validated locally with Terraform 1.14.3
(`terraform init` with no backend and no cloud credentials,
`terraform validate`, `terraform fmt -check` — all clean, AWS provider
5.100.0). Validation only checks the configuration syntax and
references; it does not provision anything.

```bash
cd terraform
terraform init
terraform fmt -check
terraform plan
```

## License

MIT — see [LICENSE](LICENSE).
