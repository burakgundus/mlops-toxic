# mlops-toxic

End-to-end toxic comment detection with a reproducible pipeline: environment setup, data versioning and validation, experiment tracking, and FastAPI serving with automated CI/CD.

Dataset: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`train.csv`, 159,571 comments, six binary labels).

## Project status

- **Setup:** Done. Poetry, pre-commit (ruff, ruff-format, mypy), Docker Compose (PostgreSQL + MLflow), profile-based config (`config/`).
- **Data:** Done. DVC versioning (local remote), Great Expectations validation, staged data layers.
- **Model:** Done. TF-IDF + Logistic Regression (one model per label), Optuna hyperparameter search, MLflow tracking and model registry (`toxic-clf`).
- **CI/CD & Testing:** Done. GitHub Actions CI pipeline (lint, format, mypy type-checking, pytest unit/integration tests).
- **Serving:** Done. FastAPI multi-label inference service.
- **Containerization:** Multi-stage API Dockerfile, Compose integration, CI container smoke check.
- **Deployment:** Registry publishing and automated deployment planned.
- **Monitoring:** Planned (Evidently, Prometheus, Grafana).

## Requirements

- Windows with Docker Desktop
- Python 3.11
- Poetry

## Setup

```powershell
git clone https://github.com/burakgundus/mlops-toxic.git
cd mlops-toxic
poetry install
poetry run pre-commit install
docker compose -f docker\docker-compose.yml up -d
```

MLflow UI: http://localhost:5000

## Data

The raw data is not stored in Git. Git keeps only the pointer file `data/01-raw/train.csv.dvc`.

Download `train.csv` from the Kaggle page above, place it at `data/01-raw/train.csv`, and check that its MD5 hash matches the one in `data/01-raw/train.csv.dvc`.

Data layers:

```text
data/01-raw/           raw input (DVC-tracked)
data/02-preprocessed/  validated and cleaned data
data/03-features/      TF-IDF features and vectorizer
data/04-predictions/   inference output
```

## Run the pipelines

Training pipeline (validate, preprocess, build features, train, register in MLflow):

```powershell
poetry run python -m entrypoint.train
```

Batch Inference (loads the registered model and writes predictions):

```powershell
poetry run python -m entrypoint.inference
```

Optuna hyperparameter search (nested MLflow runs under `optuna-search`):

```powershell
poetry run python -m src.pipelines.tuning_pipeline
```

## Run the API & Tests

Run test suite:

```powershell
poetry run pytest
```

Start FastAPI inference service:

```powershell
poetry run uvicorn src.api.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Containerized inference

From the repository root, start PostgreSQL, MLflow and the API:

```powershell
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
```

The API uses `APP_ENV=prod` and `http://mlflow:5000` on the Compose network.
The API image installs only the locked main dependency group; DVC, Great Expectations
and Optuna belong to the training group. A normal `poetry install` still installs
training and development tools. The runtime contains no Poetry or build tools and
runs as UID 10001 with one Uvicorn worker (one in-memory copy of the model).

Model loading uses `mlflow.pyfunc`, matching `ToxicClassifier` in the training pipeline.
`MODEL_URI` defaults to `models:/toxic-clf/Staging`. For a reproducible deployment,
set it to an explicit registered model version before starting Compose:

```powershell
$env:MODEL_URI = "models:/toxic-clf/1"  # Replace with an existing version.
docker compose -f docker/docker-compose.yml up -d --build api
```

A fresh PostgreSQL volume has no model registry entries. Register a trained model
before expecting a healthy API. For local training, obtain the DVC data first,
keep `APP_ENV=local`, start the backing services, then run the pipeline:

```powershell
docker compose -f docker/docker-compose.yml up -d postgres mlflow
$env:APP_ENV = "local"
poetry run python -m entrypoint.train
docker compose -f docker/docker-compose.yml restart api
```

If the API has not been created yet, use `up -d --build api` instead of `restart api`.
Loading happens once per process; restart or recreate the API after the selected
model becomes available or the Staging version changes.

```powershell
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/predict -Method Post `
  -ContentType "application/json" -Body '{"text":"Thank you for your help."}'
```

- `/health`: process liveness; includes `model_loaded`.
- `/ready`: HTTP 200 after model load, otherwise HTTP 503; used by Docker healthcheck.
- `/predict`: six probabilities, with HTTP 503 if no model is available. There are
  no mock predictions. Blank input and invalid model output are rejected.
- `/docs`: interactive API documentation.

MLflow now proxies artifacts over HTTP (`--serve-artifacts --artifacts-destination
/mlruns`), following the [MLflow artifact serving documentation](https://www.mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
Existing experiments retain their recorded artifact location. The read-only
`/mlruns` mount in the API supports existing versions whose artifacts were stored
under that absolute path. It does not repair artifacts recorded under a different
host path. For new HTTP-proxied runs, use a new experiment if the existing experiment
still points to `/mlruns`; update the selected profile's `experiment_name` first.
Do not delete existing volumes to migrate experiments.

The Compose credentials are for local development. Registry publishing, deployment
credentials and monitoring are separate follow-up work.

CI builds the API image only after the Python checks pass. It then creates a tiny
synthetic pyfunc model with runtime dependencies and checks readiness, prediction
and the non-root user. This fixture is solely a smoke test, not a trained Jigsaw model.

## Configuration

Settings live in `config/local.yaml` and `config/prod.yaml`. The profile is chosen with the `APP_ENV` environment variable (default: `local`):

```powershell
$env:APP_ENV = "prod"
```

## Project structure

```text
.github/workflows/ CI pipeline (GitHub Actions: lint, type-check, tests)
config/            environment profiles (local.yaml, prod.yaml)
data/              staged DVC data layers (01-raw to 04-predictions)
docker/            docker-compose.yml and MLflow Dockerfile
entrypoint/        train.py and inference.py entry scripts
src/api/           FastAPI application and route handlers
src/config.py      config loader
src/pipelines/     feature_eng, training, tuning and inference pipelines
tests/             unit and integration test suite
```

## Notes

- The DVC remote is currently a local folder on one machine. A shared remote is planned.
- MinIO is not part of the stack yet; MLflow stores artifacts in a Docker volume.
- Experiment results are stored in MLflow and not in Git.
